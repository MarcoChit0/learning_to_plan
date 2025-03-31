(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot20 shot61 shot298 - shot
      ingredient225 ingredient92 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser305 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot20)
  (ontable shot61)
  (ontable shot298)
  (dispenses dispenser305 ingredient225)
  (dispenses dispenser307 ingredient92)
  (clean shaker453)
  (clean shot20)
  (clean shot61)
  (clean shot298)
  (empty shaker453)
  (empty shot20)
  (empty shot61)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient92)
  (cocktail-part2 cocktail1 ingredient225)
  (cocktail-part1 cocktail2 ingredient92)
  (cocktail-part2 cocktail2 ingredient225)
)
 (:goal
  (and
      (contains shot20 cocktail1)
      (contains shot61 cocktail2)
)))
