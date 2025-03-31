; child-snack task with 4 children and 0.24 gluten factor 
; constant factor of 1.7
; random seed: 450708

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child194 child205 child374 - child
    bread210 bread269 bread78 bread114 - bread-portion
    content44 content377 content460 content128 - content-portion
    tray264 tray216 tray183 - tray
    table349 table277 table374 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 sandw7 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at tray216 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_bread bread269)
     (at_kitchen_bread bread78)
     (at_kitchen_bread bread114)
     (at_kitchen_content content44)
     (at_kitchen_content content377)
     (at_kitchen_content content460)
     (at_kitchen_content content128)
     (not_allergic_gluten child374)
     (not_allergic_gluten child205)
     (not_allergic_gluten child194)
     (not_allergic_gluten child112)
     (waiting child112 table277)
     (waiting child194 table277)
     (waiting child205 table374)
     (waiting child374 table277)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
     (notexist sandw7)
  )
  (:goal
    (and
     (served child112)
     (served child194)
     (served child205)
     (served child374)
    )
  )
)
