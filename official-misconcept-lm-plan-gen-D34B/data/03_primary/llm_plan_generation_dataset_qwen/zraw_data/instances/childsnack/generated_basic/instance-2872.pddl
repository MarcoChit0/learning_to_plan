; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 432279

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 - child
    bread75 - bread-portion
    content36 - content-portion
    tray294 tray463 - tray
    table460 table177 table168 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at tray463 kitchen)
     (at_kitchen_bread bread75)
     (at_kitchen_content content36)
     (not_allergic_gluten child450)
     (waiting child450 table177)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child450)
    )
  )
)
