; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 687292

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child76 child188 - child
    bread360 bread447 - bread-portion
    content293 content274 - content-portion
    tray333 tray103 - tray
    table343 table164 table40 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread447)
     (at_kitchen_content content293)
     (at_kitchen_content content274)
     (not_allergic_gluten child76)
     (not_allergic_gluten child188)
     (waiting child76 table343)
     (waiting child188 table164)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child76)
     (served child188)
    )
  )
)
