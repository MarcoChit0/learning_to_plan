(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot416 shot259 - shot
      ingredient471 ingredient229 ingredient217 ingredient408 - ingredient
      cocktail271 - cocktail
      dispenser243 dispenser377 dispenser395 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot416)
  (ontable shot259)
  (dispenses dispenser243 ingredient471)
  (dispenses dispenser377 ingredient229)
  (dispenses dispenser395 ingredient217)
  (dispenses dispenser92 ingredient408)
  (clean shaker489)
  (clean shot416)
  (clean shot259)
  (empty shaker489)
  (empty shot416)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient471)
  (cocktail-part2 cocktail271 ingredient217)
)
 (:goal
  (and
      (contains shot416 cocktail271)
)))
