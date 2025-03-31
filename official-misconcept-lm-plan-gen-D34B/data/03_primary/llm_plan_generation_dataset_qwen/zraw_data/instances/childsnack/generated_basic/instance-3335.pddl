; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 884627

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child198 - child
    bread413 bread439 - bread-portion
    content270 content128 - content-portion
    tray432 tray274 tray82 tray389 - tray
    table79 table401 table417 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray274 kitchen)
     (at tray82 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread439)
     (at_kitchen_content content270)
     (at_kitchen_content content128)
     (no_gluten_bread bread439)
     (no_gluten_content content270)
     (allergic_gluten child198)
     (not_allergic_gluten child349)
     (waiting child349 table401)
     (waiting child198 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child349)
     (served child198)
    )
  )
)
