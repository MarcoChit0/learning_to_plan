; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 990561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 - child
    bread226 - bread-portion
    content255 - content-portion
    tray172 tray320 tray107 - tray
    table477 table152 table44 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray172 kitchen)
     (at tray320 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_content content255)
     (not_allergic_gluten child164)
     (waiting child164 table477)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child164)
    )
  )
)
