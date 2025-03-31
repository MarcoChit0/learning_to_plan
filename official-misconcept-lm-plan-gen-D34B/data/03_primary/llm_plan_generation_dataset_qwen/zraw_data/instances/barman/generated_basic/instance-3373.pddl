(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot152 shot319 shot293 - shot
      ingredient473 ingredient92 ingredient417 - ingredient
      cocktail495 - cocktail
      dispenser425 dispenser72 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot152)
  (ontable shot319)
  (ontable shot293)
  (dispenses dispenser425 ingredient473)
  (dispenses dispenser72 ingredient92)
  (dispenses dispenser209 ingredient417)
  (clean shaker363)
  (clean shot152)
  (clean shot319)
  (clean shot293)
  (empty shaker363)
  (empty shot152)
  (empty shot319)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient417)
  (cocktail-part2 cocktail495 ingredient92)
)
 (:goal
  (and
      (contains shot152 cocktail495)
      (contains shot319 cocktail495)
)))
