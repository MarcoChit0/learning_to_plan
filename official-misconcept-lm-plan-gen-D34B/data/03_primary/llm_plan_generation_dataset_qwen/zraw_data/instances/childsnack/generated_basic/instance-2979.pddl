; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 331042

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 - child
    bread61 - bread-portion
    content163 - content-portion
    tray70 tray290 tray121 tray216 - tray
    table488 table11 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray290 kitchen)
     (at tray121 kitchen)
     (at tray216 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_content content163)
     (not_allergic_gluten child325)
     (waiting child325 table255)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child325)
    )
  )
)
