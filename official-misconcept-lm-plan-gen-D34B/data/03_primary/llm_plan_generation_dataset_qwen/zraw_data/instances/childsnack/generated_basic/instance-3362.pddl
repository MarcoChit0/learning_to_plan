; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 818512

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child186 child230 child222 - child
    bread317 bread29 bread310 - bread-portion
    content330 content16 content197 - content-portion
    tray292 tray174 tray336 - tray
    table91 table227 table96 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at tray174 kitchen)
     (at tray336 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_bread bread29)
     (at_kitchen_bread bread310)
     (at_kitchen_content content330)
     (at_kitchen_content content16)
     (at_kitchen_content content197)
     (no_gluten_bread bread29)
     (no_gluten_content content330)
     (allergic_gluten child222)
     (not_allergic_gluten child230)
     (not_allergic_gluten child186)
     (waiting child186 table227)
     (waiting child230 table227)
     (waiting child222 table227)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child186)
     (served child230)
     (served child222)
    )
  )
)
