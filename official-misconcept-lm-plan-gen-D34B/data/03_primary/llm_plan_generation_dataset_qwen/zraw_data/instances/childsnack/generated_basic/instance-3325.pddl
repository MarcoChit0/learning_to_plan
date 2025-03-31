; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 553482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child102 - child
    bread488 bread3 - bread-portion
    content3 content391 - content-portion
    tray291 - tray
    table403 table253 table39 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread3)
     (at_kitchen_content content3)
     (at_kitchen_content content391)
     (no_gluten_bread bread3)
     (no_gluten_content content391)
     (allergic_gluten child102)
     (not_allergic_gluten child417)
     (waiting child417 table253)
     (waiting child102 table39)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child417)
     (served child102)
    )
  )
)
