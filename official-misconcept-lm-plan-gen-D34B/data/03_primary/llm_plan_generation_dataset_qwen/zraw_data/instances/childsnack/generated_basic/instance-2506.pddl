; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 378309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 - child
    bread305 - bread-portion
    content370 - content-portion
    tray282 tray318 tray174 - tray
    table42 table4 table144 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray282 kitchen)
     (at tray318 kitchen)
     (at tray174 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_content content370)
     (not_allergic_gluten child171)
     (waiting child171 table42)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child171)
    )
  )
)
