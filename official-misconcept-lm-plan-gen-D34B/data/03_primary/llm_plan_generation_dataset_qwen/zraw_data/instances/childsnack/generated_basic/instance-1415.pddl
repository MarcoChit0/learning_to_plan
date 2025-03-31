; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 349411

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 child129 - child
    bread175 bread334 - bread-portion
    content260 content283 - content-portion
    tray190 - tray
    table470 table366 table428 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray190 kitchen)
     (at_kitchen_bread bread175)
     (at_kitchen_bread bread334)
     (at_kitchen_content content260)
     (at_kitchen_content content283)
     (no_gluten_bread bread334)
     (no_gluten_content content283)
     (allergic_gluten child129)
     (not_allergic_gluten child499)
     (waiting child499 table366)
     (waiting child129 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child499)
     (served child129)
    )
  )
)
