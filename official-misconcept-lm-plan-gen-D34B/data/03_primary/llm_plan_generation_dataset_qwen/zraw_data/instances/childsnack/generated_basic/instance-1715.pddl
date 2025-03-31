; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 471562

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 - child
    bread350 - bread-portion
    content130 - content-portion
    tray289 tray228 tray120 - tray
    table286 table144 table155 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray228 kitchen)
     (at tray120 kitchen)
     (at_kitchen_bread bread350)
     (at_kitchen_content content130)
     (not_allergic_gluten child446)
     (waiting child446 table155)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child446)
    )
  )
)
