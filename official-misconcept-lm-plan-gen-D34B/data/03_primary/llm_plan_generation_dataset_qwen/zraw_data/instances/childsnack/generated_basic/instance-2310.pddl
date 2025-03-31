; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 837688

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 - child
    bread210 - bread-portion
    content455 - content-portion
    tray212 - tray
    table364 table213 table65 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_content content455)
     (not_allergic_gluten child35)
     (waiting child35 table364)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child35)
    )
  )
)
