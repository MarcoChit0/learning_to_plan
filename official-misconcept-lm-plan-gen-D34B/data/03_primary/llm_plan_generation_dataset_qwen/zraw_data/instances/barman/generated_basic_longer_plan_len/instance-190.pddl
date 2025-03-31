(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot446 shot481 shot361 - shot
      ingredient48 ingredient338 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser207 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot446)
  (ontable shot481)
  (ontable shot361)
  (dispenses dispenser207 ingredient48)
  (dispenses dispenser132 ingredient338)
  (clean shaker103)
  (clean shot446)
  (clean shot481)
  (clean shot361)
  (empty shaker103)
  (empty shot446)
  (empty shot481)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient338)
  (cocktail-part1 cocktail2 ingredient338)
  (cocktail-part2 cocktail2 ingredient48)
)
 (:goal
  (and
      (contains shot446 cocktail1)
      (contains shot481 cocktail2)
)))
