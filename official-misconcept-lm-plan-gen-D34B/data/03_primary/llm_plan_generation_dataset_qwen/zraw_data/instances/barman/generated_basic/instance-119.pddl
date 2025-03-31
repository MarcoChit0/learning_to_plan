(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot437 shot26 - shot
      ingredient443 ingredient205 - ingredient
      cocktail1 - cocktail
      dispenser142 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot437)
  (ontable shot26)
  (dispenses dispenser142 ingredient443)
  (dispenses dispenser492 ingredient205)
  (clean shaker174)
  (clean shot437)
  (clean shot26)
  (empty shaker174)
  (empty shot437)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient443)
  (cocktail-part2 cocktail1 ingredient205)
)
 (:goal
  (and
      (contains shot437 cocktail1)
)))
