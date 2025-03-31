(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot30 shot285 - shot
      ingredient119 ingredient184 ingredient355 - ingredient
      cocktail365 - cocktail
      dispenser178 dispenser40 dispenser334 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot30)
  (ontable shot285)
  (dispenses dispenser178 ingredient119)
  (dispenses dispenser40 ingredient184)
  (dispenses dispenser334 ingredient355)
  (clean shaker83)
  (clean shot30)
  (clean shot285)
  (empty shaker83)
  (empty shot30)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient184)
  (cocktail-part2 cocktail365 ingredient355)
)
 (:goal
  (and
      (contains shot30 cocktail365)
)))
