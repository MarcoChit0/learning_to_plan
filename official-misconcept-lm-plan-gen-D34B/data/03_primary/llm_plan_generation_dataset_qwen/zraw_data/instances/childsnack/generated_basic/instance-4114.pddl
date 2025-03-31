; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 276215

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child269 child302 - child
    bread454 bread226 - bread-portion
    content282 content362 - content-portion
    tray465 tray384 tray453 - tray
    table182 table312 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at tray384 kitchen)
     (at tray453 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread226)
     (at_kitchen_content content282)
     (at_kitchen_content content362)
     (not_allergic_gluten child269)
     (not_allergic_gluten child302)
     (waiting child269 table182)
     (waiting child302 table182)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child269)
     (served child302)
    )
  )
)
