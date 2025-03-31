(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot230 shot460 shot353 - shot
      ingredient187 ingredient41 ingredient195 ingredient5 - ingredient
      cocktail142 - cocktail
      dispenser471 dispenser148 dispenser280 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot230)
  (ontable shot460)
  (ontable shot353)
  (dispenses dispenser471 ingredient187)
  (dispenses dispenser148 ingredient41)
  (dispenses dispenser280 ingredient195)
  (dispenses dispenser129 ingredient5)
  (clean shaker192)
  (clean shot230)
  (clean shot460)
  (clean shot353)
  (empty shaker192)
  (empty shot230)
  (empty shot460)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient5)
  (cocktail-part2 cocktail142 ingredient187)
)
 (:goal
  (and
      (contains shot230 cocktail142)
      (contains shot460 cocktail142)
)))
