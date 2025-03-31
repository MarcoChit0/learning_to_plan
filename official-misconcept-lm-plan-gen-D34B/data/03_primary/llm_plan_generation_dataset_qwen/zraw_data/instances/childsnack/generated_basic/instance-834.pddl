; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 971554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child277 - child
    bread297 bread88 - bread-portion
    content298 content365 - content-portion
    tray187 tray60 - tray
    table232 table45 table38 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray187 kitchen)
     (at tray60 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread88)
     (at_kitchen_content content298)
     (at_kitchen_content content365)
     (no_gluten_bread bread297)
     (no_gluten_content content365)
     (allergic_gluten child332)
     (not_allergic_gluten child277)
     (waiting child332 table232)
     (waiting child277 table38)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child332)
     (served child277)
    )
  )
)
