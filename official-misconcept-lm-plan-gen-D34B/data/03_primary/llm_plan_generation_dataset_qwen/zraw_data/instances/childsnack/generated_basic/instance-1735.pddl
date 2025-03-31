; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 566100

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 - child
    bread127 - bread-portion
    content291 - content-portion
    tray232 tray97 tray21 - tray
    table331 table207 table252 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray232 kitchen)
     (at tray97 kitchen)
     (at tray21 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_content content291)
     (not_allergic_gluten child367)
     (waiting child367 table207)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child367)
    )
  )
)
