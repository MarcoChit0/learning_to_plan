(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot17 shot143 shot270 - shot
      ingredient353 ingredient318 ingredient370 ingredient438 - ingredient
      cocktail346 - cocktail
      dispenser483 dispenser244 dispenser147 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot17)
  (ontable shot143)
  (ontable shot270)
  (dispenses dispenser483 ingredient353)
  (dispenses dispenser244 ingredient318)
  (dispenses dispenser147 ingredient370)
  (dispenses dispenser66 ingredient438)
  (clean shaker4)
  (clean shot17)
  (clean shot143)
  (clean shot270)
  (empty shaker4)
  (empty shot17)
  (empty shot143)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient438)
  (cocktail-part2 cocktail346 ingredient370)
)
 (:goal
  (and
      (contains shot17 cocktail346)
      (contains shot143 cocktail346)
)))
