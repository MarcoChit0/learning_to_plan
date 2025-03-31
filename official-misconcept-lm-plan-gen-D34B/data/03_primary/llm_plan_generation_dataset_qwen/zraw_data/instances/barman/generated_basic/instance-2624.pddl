(define (problem prob)
 (:domain barman)
 (:objects 
      shaker346 - shaker
      left right - hand
      shot372 shot260 - shot
      ingredient371 ingredient219 ingredient260 - ingredient
      cocktail155 - cocktail
      dispenser240 dispenser57 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker346)
  (ontable shot372)
  (ontable shot260)
  (dispenses dispenser240 ingredient371)
  (dispenses dispenser57 ingredient219)
  (dispenses dispenser23 ingredient260)
  (clean shaker346)
  (clean shot372)
  (clean shot260)
  (empty shaker346)
  (empty shot372)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker346 l0)
  (shaker-level shaker346 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient219)
  (cocktail-part2 cocktail155 ingredient371)
)
 (:goal
  (and
      (contains shot372 cocktail155)
)))
