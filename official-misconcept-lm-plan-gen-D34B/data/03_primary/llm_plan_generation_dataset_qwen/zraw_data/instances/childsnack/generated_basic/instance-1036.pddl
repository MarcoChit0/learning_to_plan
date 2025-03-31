; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 463271

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 child277 - child
    bread1 bread280 - bread-portion
    content162 content38 - content-portion
    tray499 tray169 tray48 - tray
    table229 table351 table396 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray169 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread1)
     (at_kitchen_bread bread280)
     (at_kitchen_content content162)
     (at_kitchen_content content38)
     (not_allergic_gluten child291)
     (not_allergic_gluten child277)
     (waiting child291 table396)
     (waiting child277 table229)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child291)
     (served child277)
    )
  )
)
