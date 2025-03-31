(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot192 shot85 shot174 - shot
      ingredient289 ingredient34 ingredient285 ingredient260 - ingredient
      cocktail1 - cocktail
      dispenser354 dispenser169 dispenser115 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot192)
  (ontable shot85)
  (ontable shot174)
  (dispenses dispenser354 ingredient289)
  (dispenses dispenser169 ingredient34)
  (dispenses dispenser115 ingredient285)
  (dispenses dispenser425 ingredient260)
  (clean shaker95)
  (clean shot192)
  (clean shot85)
  (clean shot174)
  (empty shaker95)
  (empty shot192)
  (empty shot85)
  (empty shot174)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient260)
  (cocktail-part2 cocktail1 ingredient289)
)
 (:goal
  (and
      (contains shot192 cocktail1)
      (contains shot85 cocktail1)
)))
