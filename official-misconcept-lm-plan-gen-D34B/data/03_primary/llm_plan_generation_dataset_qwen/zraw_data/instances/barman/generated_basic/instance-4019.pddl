(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot89 shot319 - shot
      ingredient366 ingredient495 ingredient103 - ingredient
      cocktail361 - cocktail
      dispenser304 dispenser313 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot89)
  (ontable shot319)
  (dispenses dispenser304 ingredient366)
  (dispenses dispenser313 ingredient495)
  (dispenses dispenser263 ingredient103)
  (clean shaker257)
  (clean shot89)
  (clean shot319)
  (empty shaker257)
  (empty shot89)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail361 ingredient495)
  (cocktail-part2 cocktail361 ingredient366)
)
 (:goal
  (and
      (contains shot89 cocktail361)
)))
