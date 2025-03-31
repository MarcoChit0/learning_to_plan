; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 715151

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 - child
    bread148 - bread-portion
    content175 - content-portion
    tray169 tray455 tray164 - tray
    table177 table327 table197 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray455 kitchen)
     (at tray164 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_content content175)
     (not_allergic_gluten child353)
     (waiting child353 table327)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child353)
    )
  )
)
