(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot59 shot460 shot440 - shot
      ingredient16 ingredient106 ingredient157 ingredient393 - ingredient
      cocktail1 - cocktail
      dispenser415 dispenser189 dispenser127 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot59)
  (ontable shot460)
  (ontable shot440)
  (dispenses dispenser415 ingredient16)
  (dispenses dispenser189 ingredient106)
  (dispenses dispenser127 ingredient157)
  (dispenses dispenser400 ingredient393)
  (clean shaker220)
  (clean shot59)
  (clean shot460)
  (clean shot440)
  (empty shaker220)
  (empty shot59)
  (empty shot460)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient157)
  (cocktail-part2 cocktail1 ingredient106)
)
 (:goal
  (and
      (contains shot59 cocktail1)
      (contains shot460 cocktail1)
)))
