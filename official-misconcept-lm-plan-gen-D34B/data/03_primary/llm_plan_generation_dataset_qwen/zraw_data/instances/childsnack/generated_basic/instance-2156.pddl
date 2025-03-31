; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 678341

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 - child
    bread18 - bread-portion
    content465 - content-portion
    tray418 - tray
    table258 table36 table166 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_content content465)
     (not_allergic_gluten child78)
     (waiting child78 table166)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child78)
    )
  )
)
