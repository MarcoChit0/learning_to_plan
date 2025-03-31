; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 520182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 child380 - child
    bread186 bread269 - bread-portion
    content362 content38 - content-portion
    tray298 - tray
    table92 table178 table121 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at_kitchen_bread bread186)
     (at_kitchen_bread bread269)
     (at_kitchen_content content362)
     (at_kitchen_content content38)
     (no_gluten_bread bread186)
     (no_gluten_content content38)
     (allergic_gluten child304)
     (not_allergic_gluten child380)
     (waiting child304 table178)
     (waiting child380 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child304)
     (served child380)
    )
  )
)
