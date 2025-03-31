(define (problem prob)
 (:domain barman)
 (:objects 
      shaker289 - shaker
      left right - hand
      shot185 shot212 shot309 - shot
      ingredient203 ingredient246 ingredient493 - ingredient
      cocktail186 - cocktail
      dispenser360 dispenser181 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker289)
  (ontable shot185)
  (ontable shot212)
  (ontable shot309)
  (dispenses dispenser360 ingredient203)
  (dispenses dispenser181 ingredient246)
  (dispenses dispenser32 ingredient493)
  (clean shaker289)
  (clean shot185)
  (clean shot212)
  (clean shot309)
  (empty shaker289)
  (empty shot185)
  (empty shot212)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker289 l0)
  (shaker-level shaker289 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient246)
  (cocktail-part2 cocktail186 ingredient493)
)
 (:goal
  (and
      (contains shot185 cocktail186)
      (contains shot212 ingredient493)
)))
