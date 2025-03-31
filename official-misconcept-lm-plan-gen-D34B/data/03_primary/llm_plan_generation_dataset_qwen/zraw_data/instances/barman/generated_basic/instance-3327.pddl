(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot166 shot466 shot124 - shot
      ingredient21 ingredient404 ingredient165 - ingredient
      cocktail130 - cocktail
      dispenser132 dispenser15 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot166)
  (ontable shot466)
  (ontable shot124)
  (dispenses dispenser132 ingredient21)
  (dispenses dispenser15 ingredient404)
  (dispenses dispenser275 ingredient165)
  (clean shaker116)
  (clean shot166)
  (clean shot466)
  (clean shot124)
  (empty shaker116)
  (empty shot166)
  (empty shot466)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail130 ingredient21)
  (cocktail-part2 cocktail130 ingredient165)
)
 (:goal
  (and
      (contains shot166 cocktail130)
      (contains shot466 cocktail130)
)))
