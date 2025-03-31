; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 261012

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 child287 - child
    bread293 bread295 - bread-portion
    content89 content174 - content-portion
    tray419 - tray
    table403 table193 table219 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray419 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_bread bread295)
     (at_kitchen_content content89)
     (at_kitchen_content content174)
     (no_gluten_bread bread295)
     (no_gluten_content content174)
     (allergic_gluten child267)
     (not_allergic_gluten child287)
     (waiting child267 table219)
     (waiting child287 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child267)
     (served child287)
    )
  )
)
