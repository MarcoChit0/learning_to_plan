(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot130 shot124 - shot
      ingredient268 ingredient221 ingredient365 ingredient51 - ingredient
      cocktail166 - cocktail
      dispenser364 dispenser355 dispenser181 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot130)
  (ontable shot124)
  (dispenses dispenser364 ingredient268)
  (dispenses dispenser355 ingredient221)
  (dispenses dispenser181 ingredient365)
  (dispenses dispenser278 ingredient51)
  (clean shaker241)
  (clean shot130)
  (clean shot124)
  (empty shaker241)
  (empty shot130)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail166 ingredient365)
  (cocktail-part2 cocktail166 ingredient221)
)
 (:goal
  (and
      (contains shot130 cocktail166)
)))
