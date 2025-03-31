; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 952604

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child190 - child
    bread440 - bread-portion
    content346 - content-portion
    tray335 tray11 tray360 - tray
    table383 table190 table325 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray11 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_content content346)
     (not_allergic_gluten child190)
     (waiting child190 table190)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child190)
    )
  )
)
