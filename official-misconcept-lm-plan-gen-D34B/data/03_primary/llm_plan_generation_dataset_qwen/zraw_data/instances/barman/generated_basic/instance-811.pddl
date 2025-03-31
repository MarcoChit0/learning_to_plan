(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot295 shot70 - shot
      ingredient395 ingredient126 ingredient332 ingredient62 - ingredient
      cocktail1 - cocktail
      dispenser272 dispenser211 dispenser96 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot295)
  (ontable shot70)
  (dispenses dispenser272 ingredient395)
  (dispenses dispenser211 ingredient126)
  (dispenses dispenser96 ingredient332)
  (dispenses dispenser356 ingredient62)
  (clean shaker446)
  (clean shot295)
  (clean shot70)
  (empty shaker446)
  (empty shot295)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient395)
  (cocktail-part2 cocktail1 ingredient126)
)
 (:goal
  (and
      (contains shot295 cocktail1)
)))
