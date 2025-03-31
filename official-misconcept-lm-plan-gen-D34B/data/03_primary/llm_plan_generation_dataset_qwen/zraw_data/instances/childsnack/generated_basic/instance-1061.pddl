; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 855402

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 child42 - child
    bread326 bread307 - bread-portion
    content405 content215 - content-portion
    tray245 tray463 tray350 - tray
    table284 table239 table43 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray463 kitchen)
     (at tray350 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread307)
     (at_kitchen_content content405)
     (at_kitchen_content content215)
     (not_allergic_gluten child198)
     (not_allergic_gluten child42)
     (waiting child198 table284)
     (waiting child42 table239)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child198)
     (served child42)
    )
  )
)
