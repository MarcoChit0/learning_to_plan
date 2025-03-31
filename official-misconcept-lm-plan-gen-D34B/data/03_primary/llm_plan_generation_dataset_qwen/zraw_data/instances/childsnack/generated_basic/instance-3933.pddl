; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 797338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child179 - child
    bread34 bread464 - bread-portion
    content244 content48 - content-portion
    tray321 - tray
    table317 table287 table229 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread464)
     (at_kitchen_content content244)
     (at_kitchen_content content48)
     (no_gluten_bread bread464)
     (no_gluten_content content48)
     (allergic_gluten child202)
     (not_allergic_gluten child179)
     (waiting child202 table287)
     (waiting child179 table317)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child202)
     (served child179)
    )
  )
)
