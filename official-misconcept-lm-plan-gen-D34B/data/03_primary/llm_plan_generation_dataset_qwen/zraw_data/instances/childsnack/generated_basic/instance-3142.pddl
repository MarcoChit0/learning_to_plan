; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 549199

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 child75 - child
    bread366 bread424 - bread-portion
    content411 content244 - content-portion
    tray93 tray285 - tray
    table308 table455 table4 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread366)
     (at_kitchen_bread bread424)
     (at_kitchen_content content411)
     (at_kitchen_content content244)
     (not_allergic_gluten child75)
     (not_allergic_gluten child198)
     (waiting child198 table308)
     (waiting child75 table455)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child198)
     (served child75)
    )
  )
)
