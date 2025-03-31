; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 494579

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child241 - child
    bread132 - bread-portion
    content485 - content-portion
    tray481 tray225 tray357 tray235 - tray
    table377 table82 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray225 kitchen)
     (at tray357 kitchen)
     (at tray235 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_content content485)
     (not_allergic_gluten child241)
     (waiting child241 table377)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child241)
    )
  )
)
