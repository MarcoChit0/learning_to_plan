; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 415571

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 - child
    bread469 - bread-portion
    content435 - content-portion
    tray114 tray149 tray145 - tray
    table415 table14 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray149 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread469)
     (at_kitchen_content content435)
     (not_allergic_gluten child274)
     (waiting child274 table415)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child274)
    )
  )
)
