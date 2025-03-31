(define (problem prob)
 (:domain barman)
 (:objects 
      shaker455 - shaker
      left right - hand
      shot173 shot208 - shot
      ingredient321 ingredient389 ingredient397 ingredient320 - ingredient
      cocktail1 - cocktail
      dispenser355 dispenser368 dispenser338 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker455)
  (ontable shot173)
  (ontable shot208)
  (dispenses dispenser355 ingredient321)
  (dispenses dispenser368 ingredient389)
  (dispenses dispenser338 ingredient397)
  (dispenses dispenser60 ingredient320)
  (clean shaker455)
  (clean shot173)
  (clean shot208)
  (empty shaker455)
  (empty shot173)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker455 l0)
  (shaker-level shaker455 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient320)
)
 (:goal
  (and
      (contains shot173 cocktail1)
)))
