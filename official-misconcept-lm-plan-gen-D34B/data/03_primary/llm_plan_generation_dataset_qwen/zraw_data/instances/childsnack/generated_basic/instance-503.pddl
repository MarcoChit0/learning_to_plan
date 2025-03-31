; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 373231

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 child122 - child
    bread343 bread223 - bread-portion
    content375 content459 - content-portion
    tray186 tray407 - tray
    table84 table187 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread223)
     (at_kitchen_content content375)
     (at_kitchen_content content459)
     (not_allergic_gluten child305)
     (not_allergic_gluten child122)
     (waiting child305 table193)
     (waiting child122 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child305)
     (served child122)
    )
  )
)
