; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 280669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 - child
    bread305 - bread-portion
    content210 - content-portion
    tray192 tray235 tray369 tray406 - tray
    table99 table149 table97 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at tray235 kitchen)
     (at tray369 kitchen)
     (at tray406 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_content content210)
     (not_allergic_gluten child481)
     (waiting child481 table97)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child481)
    )
  )
)
