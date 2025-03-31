; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 645627

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread496 - bread-portion
    content346 - content-portion
    tray185 tray263 tray282 tray245 - tray
    table325 table282 table368 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray263 kitchen)
     (at tray282 kitchen)
     (at tray245 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_content content346)
     (not_allergic_gluten child33)
     (waiting child33 table282)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
