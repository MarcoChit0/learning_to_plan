; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 389074

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 - child
    bread331 - bread-portion
    content446 - content-portion
    tray99 tray278 tray84 - tray
    table67 table421 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray278 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content446)
     (not_allergic_gluten child381)
     (waiting child381 table67)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child381)
    )
  )
)
