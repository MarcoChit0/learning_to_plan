; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 243702

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 - child
    bread93 - bread-portion
    content371 - content-portion
    tray483 tray476 tray404 - tray
    table27 table220 table259 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray476 kitchen)
     (at tray404 kitchen)
     (at_kitchen_bread bread93)
     (at_kitchen_content content371)
     (not_allergic_gluten child3)
     (waiting child3 table259)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child3)
    )
  )
)
