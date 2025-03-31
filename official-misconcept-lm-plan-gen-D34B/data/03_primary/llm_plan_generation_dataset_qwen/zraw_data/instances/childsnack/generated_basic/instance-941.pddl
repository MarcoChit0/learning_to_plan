; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 761305

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child472 child340 child64 - child
    bread326 bread330 bread458 - bread-portion
    content415 content403 content469 - content-portion
    tray0 tray253 tray417 tray22 - tray
    table358 table64 table336 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray0 kitchen)
     (at tray253 kitchen)
     (at tray417 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread330)
     (at_kitchen_bread bread458)
     (at_kitchen_content content415)
     (at_kitchen_content content403)
     (at_kitchen_content content469)
     (not_allergic_gluten child472)
     (not_allergic_gluten child340)
     (not_allergic_gluten child64)
     (waiting child472 table64)
     (waiting child340 table336)
     (waiting child64 table64)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child472)
     (served child340)
     (served child64)
    )
  )
)
