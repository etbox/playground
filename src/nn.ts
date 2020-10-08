/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  readonly id: string;
  /** List of input links. */
  readonly inputLinks: Link[] = [];
  private bias = 0.1;
  /** List of output links. */
  readonly outputLinks: Link[] = [];
  private totalInput: number;
  private output: number;
  /** Error derivative with respect to this node's output. */
  private outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  private inputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  private accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  private numAccDers = 0;
  /** Activation function that takes total input and returns node's output */
  readonly activation: ActivationFunction;

  /** Creates a new node with the provided id and activation function. */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput() {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      const link = this.inputLinks[j];
      const wj_aj = link.getWeight() * link.source.output;
      this.setTotalInput(this.getTotalInput() + wj_aj);
    }
    this.output = this.activation.output(this.totalInput);
  }

  addInputLink(inputLink: Link) {
    this.inputLinks.push(inputLink);
  }
  addOutputLink(outputLink: Link) {
    this.outputLinks.push(outputLink);
  }
  initOutput(input: number) {
    this.output = input;
  }
  getOutput(): number {
    return this.output;
  }
  setOutputDer(outputDer: number) {
    this.outputDer = outputDer;
  }
  getOutputDer(): number {
    return this.outputDer;
  }
  setInputDer(inputDer: number) {
    this.inputDer = inputDer;
  }
  getInputDer(): number {
    return this.inputDer;
  }
  setAccInputDer(accInputDer: number) {
    this.accInputDer = accInputDer;
  }
  getAccInputDer() {
    return this.accInputDer;
  }
  setNumAccDers(numAccDers: number) {
    this.numAccDers = numAccDers;
  }
  getNumAccDers(): number {
    return this.numAccDers
  }
  setTotalInput(totalInput: number) {
    this.totalInput = totalInput;
  }
  getTotalInput(): number {
    return this.totalInput
  }
  setBias(bias: number) {
    this.bias = bias;
  }
  getBias(): number {
    return this.bias
  }
}

/** An error function and its derivative. */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      const output = Activations.TANH.output(x);
      return 1 - output * output;
    }
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      const output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    }
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1
  };
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  readonly id: string;
  readonly source: Node;
  readonly dest: Node;
  private weight = Math.random() - 0.5;
  isDead = false;
  // /** Error derivative with respect to this weight. */
  // private errorDer = 0;
  /** Accumulated error derivative since the last update. */
  private accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  private numAccDers = 0;
  readonly regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node,
      regularization: RegularizationFunction, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }

  // setErrorDer(errorDer: number) {
  //   this.errorDer = errorDer;
  // }
  setAccErrorDer(accErrorDer: number) {
    this.accErrorDer = accErrorDer;
  }
  getAccErrorDer(): number {
    return this.accErrorDer;
  }
  setNumAccDers(numAccDers: number) {
    this.numAccDers = numAccDers;
  }
  getNumAccDers(): number {
    return this.numAccDers;
  }
  setWeight(weight: number) {
    this.weight = weight;
  }
  getWeight(): number {
    return this.weight;
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    regularization: RegularizationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
  const numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  const network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const isOutputLayer = layerIdx === numLayers - 1;
    const isInputLayer = layerIdx === 0;
    const currentLayer: Node[] = [];
    network.push(currentLayer);
    const numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      const node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          const prevNode = network[layerIdx - 1][j];
          const link = new Link(prevNode, node, regularization, initZero);
          prevNode.addOutputLink(link);
          node.addInputLink(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  const inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    const node = inputLayer[i];
    node.initOutput(inputs[i])
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].getOutput();
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 *
 * @param network The neural network.
 * @param target The provided target for computing error.
 * @param errorFunc The user-defined error function for computing error.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  const outputNode = network[network.length - 1][0];
  const lastOutputDer = errorFunc.der(outputNode.getOutput(), target);
  outputNode.setOutputDer(lastOutputDer);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    const currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      const inputDer = node.getOutputDer() * node.activation.der(node.getTotalInput());
      node.setInputDer(inputDer);
      node.setAccInputDer(node.getAccInputDer() + inputDer);
      node.setNumAccDers(node.getNumAccDers() + 1);
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        const errorDer = node.getInputDer() * link.source.getOutput();
        // link.setErrorDer(errorDer);
        link.setAccErrorDer(link.getAccErrorDer() + errorDer);
        link.setNumAccDers(link.getNumAccDers() + 1);
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    const prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      const node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.setOutputDer(0);
      for (let j = 0; j < node.outputLinks.length; j++) {
        const outputLink = node.outputLinks[j];
        const outputDer = node.getOutputDer() + outputLink.getWeight() * outputLink.dest.getInputDer();
        node.setOutputDer(outputDer);
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularizationRate: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      // Update the node's bias.
      if (node.getNumAccDers() > 0) {
        const averageAccDer = node.getAccInputDer() / node.getNumAccDers();
        const decrement = learningRate * averageAccDer;
        node.setBias(node.getBias() - decrement);
        node.setAccInputDer(0);
        node.setNumAccDers(0);
      }
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        const regulDer = link.regularization ?
            link.regularization.der(link.getWeight()) : 0;
        if (link.getNumAccDers() > 0) {
          // Update the weight based on dE/dw.
          const dE_dw = (learningRate / link.getNumAccDers()) * link.getAccErrorDer();
          link.setWeight(link.getWeight() - dE_dw);
          // Further update the weight based on regularization.
          const newLinkWeight = link.getWeight() -
              (learningRate * regularizationRate) * regulDer;
          if (link.regularization === RegularizationFunction.L1 &&
              link.getWeight() * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.setWeight(0);
            link.isDead = true;
          } else {
            link.setWeight(newLinkWeight);
          }
          link.setAccErrorDer(0);
          link.setNumAccDers(0);
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    const currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
