; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 785123

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 child414 - child
    bread380 bread237 - bread-portion
    content246 content244 - content-portion
    tray132 tray169 tray336 tray236 - tray
    table329 table450 table383 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at tray169 kitchen)
     (at tray336 kitchen)
     (at tray236 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_bread bread237)
     (at_kitchen_content content246)
     (at_kitchen_content content244)
     (no_gluten_bread bread237)
     (no_gluten_content content246)
     (allergic_gluten child414)
     (not_allergic_gluten child373)
     (waiting child373 table329)
     (waiting child414 table383)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child373)
     (served child414)
    )
  )
)
