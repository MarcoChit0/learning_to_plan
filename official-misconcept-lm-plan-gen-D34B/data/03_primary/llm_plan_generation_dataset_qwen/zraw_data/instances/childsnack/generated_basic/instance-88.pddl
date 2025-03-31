; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 63819

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child206 child190 - child
    bread327 bread366 - bread-portion
    content358 content64 - content-portion
    tray64 tray462 - tray
    table166 table472 table254 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray64 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_bread bread366)
     (at_kitchen_content content358)
     (at_kitchen_content content64)
     (not_allergic_gluten child190)
     (not_allergic_gluten child206)
     (waiting child206 table472)
     (waiting child190 table166)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child206)
     (served child190)
    )
  )
)
