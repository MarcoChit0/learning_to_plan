(define (problem prob)
 (:domain barman)
 (:objects 
      shaker60 - shaker
      left right - hand
      shot416 shot296 - shot
      ingredient210 ingredient16 ingredient100 - ingredient
      cocktail371 - cocktail
      dispenser471 dispenser37 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker60)
  (ontable shot416)
  (ontable shot296)
  (dispenses dispenser471 ingredient210)
  (dispenses dispenser37 ingredient16)
  (dispenses dispenser188 ingredient100)
  (clean shaker60)
  (clean shot416)
  (clean shot296)
  (empty shaker60)
  (empty shot416)
  (empty shot296)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker60 l0)
  (shaker-level shaker60 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient100)
  (cocktail-part2 cocktail371 ingredient210)
)
 (:goal
  (and
      (contains shot416 cocktail371)
)))
