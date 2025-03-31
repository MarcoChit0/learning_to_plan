(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot493 shot137 shot139 - shot
      ingredient407 ingredient164 ingredient307 ingredient196 - ingredient
      cocktail1 - cocktail
      dispenser431 dispenser110 dispenser23 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot493)
  (ontable shot137)
  (ontable shot139)
  (dispenses dispenser431 ingredient407)
  (dispenses dispenser110 ingredient164)
  (dispenses dispenser23 ingredient307)
  (dispenses dispenser467 ingredient196)
  (clean shaker101)
  (clean shot493)
  (clean shot137)
  (clean shot139)
  (empty shaker101)
  (empty shot493)
  (empty shot137)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient164)
  (cocktail-part2 cocktail1 ingredient307)
)
 (:goal
  (and
      (contains shot493 cocktail1)
      (contains shot137 cocktail1)
)))
