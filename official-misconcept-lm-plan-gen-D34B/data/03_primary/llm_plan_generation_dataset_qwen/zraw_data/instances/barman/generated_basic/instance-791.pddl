(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot10 - shot
      ingredient399 ingredient278 - ingredient
      cocktail1 - cocktail
      dispenser467 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot10)
  (dispenses dispenser467 ingredient399)
  (dispenses dispenser203 ingredient278)
  (clean shaker474)
  (clean shot10)
  (empty shaker474)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient399)
)
 (:goal
  (and
      (contains shot10 cocktail1)
)))
